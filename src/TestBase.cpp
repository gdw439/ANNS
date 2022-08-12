
#include <omp.h> 
#include <bits/stdc++.h>
#include "UtilTools.hpp"
#include "hnswlib/hnswlib.h"

#include "faiss/IndexIVF.h"
#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/impl/io.h"
#include "faiss/index_io.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/impl/AuxIndexStructures.h"

using namespace std;
using namespace utiltools;

using idx_t = hnswlib::labeltype;
using Index = hnswlib::AlgorithmInterface<float>;

using QueryVec = vector<vector<float>>;
using TopKReCall = vector<vector<pair<float, idx_t>>>;
using FaissIndex = faiss::IndexIVFFlat;


FaissIndex* TransIndex2IndexPtr(FaissIndex &index) {
    faiss::VectorIOWriter indexWriter;
    faiss::write_index(&index, &indexWriter);

    faiss::VectorIOReader indexReader;
    indexReader.data = indexWriter.data;
    FaissIndex* newIndexPtr = dynamic_cast<FaissIndex*>(faiss::read_index(&indexReader));
    return newIndexPtr;
}

Index* BuildBruteIndex(QueryVec &dataVec, hnswlib::InnerProductSpace &space) {
    if (dataVec.size() == 0) { cout << "Empty Data" << endl; return NULL; }
    cout << "dim: " << dataVec[0].size() << endl;
    Index* index = new hnswlib::BruteforceSearch<float>(&space, 2 * dataVec.size());
    for (idx_t i = 0; i < dataVec.size(); ++i) {
        index->addPoint(dataVec[i].data(), i);
    }
    return index;
}

Index* BuildHnswIndex(QueryVec &dataVec, hnswlib::InnerProductSpace &space) {
    int M = 16, efConstruction = 200;
    if (dataVec.size() == 0) { cout << "Empty Data" << endl; return NULL; }
    Index* index = new hnswlib::HierarchicalNSW<float>(&space, dataVec.size() + 1, M, efConstruction);

    #pragma omp parallel for
    for (idx_t i = 0; i < dataVec.size(); ++i) {
        index->addPoint(dataVec[i].data(), i);
    }
    return index;
}

FaissIndex* BuildFaissIndex(QueryVec &dataVec) {
    if (dataVec.size() == 0) { cout << "Empty Data" << endl; return NULL; }
    const long lens = dataVec.size();
    const long dims = dataVec[0].size();

    idx_t *index_id = new idx_t[lens];
    float *feature = new float[lens * dims];
    memset(index_id, 0, lens * sizeof(long));
    memset(feature, 0, lens * dims * sizeof(float));

    for (idx_t i = 0; i < lens; ++i) {
        index_id[i] = i;
        copy(dataVec[i].begin(), dataVec[i].end(), &feature[i * dims]);
    }

    // 参数的设置参考检索问答
    faiss::IndexFlatL2 quantizer(dims);
    // long nList = 1;
    long nList = (lens > 10000) ? (4 * sqrt(lens)) : 1;
    FaissIndex faissIndex(&quantizer, dims, nList, faiss::METRIC_INNER_PRODUCT);
    faissIndex.train(lens, feature);
    faissIndex.add_with_ids(lens, feature, index_id);

    delete [] index_id, feature;
    return TransIndex2IndexPtr(faissIndex);
}

void SearchKCloserFirst(Index* index, const size_t topK, QueryVec& dataVec, TopKReCall& reCallVec) {
    if (NULL == index) { cout << "NULL of Index Error" << endl; return; }
    for (int i = 0; i < dataVec.size(); ++ i) {
        const void* p = dataVec[i].data();
        auto resp = index->searchKnnCloserFirst(p, topK);
        reCallVec.push_back(resp);
    }
}

void SearchKCloserFirst(FaissIndex* faissIndex, const size_t topK, QueryVec& dataVec, TopKReCall& reCallVec) {
    if (NULL == faissIndex) { cout << "NULL of Index Error" << endl; return; }
    faissIndex->nprobe = (faissIndex->nprobe <= 0) ? (faissIndex->nlist / 10 + 1) : 1;
    cout << "nprobe:" << faissIndex->nprobe << " nList:" << faissIndex->nlist << endl;
    faissIndex->nprobe = (faissIndex->nprobe <= 0) ? (faissIndex->nlist / 10 + 1) : faissIndex->nprobe;

    for (int i = 0; i < dataVec.size(); ++ i) { 
        float score[topK]; idx_t index[topK];
        faissIndex->search(1, dataVec[i].data(), topK, score, index);

        vector<pair<float, idx_t>> resp(topK);
        for (int i = 0; i < topK; ++i) resp[i] = make_pair(score[i], index[i]);
        reCallVec.push_back(resp);
    }
}


void LoadRepoData(string filePath, int dim, int repoSize, int querySize, 
    vector<vector<float>>& repoVec, vector<vector<float>>& queryVec) {
    ifstream reader(filePath, ifstream::binary);

    // 加载索引库向量
    for (int i = 0; i < repoSize; ++i) {
        int length = 0;
        if (reader.read((char *)&length, 1 * sizeof(int)).gcount() == 0) {
            cout << "Load Failed" << endl;
            reader.close();
            return;
        }
        char *text = new char[length + 1];
        memset(text, 0, length + 1);
        if (reader.read(text, length).gcount() == 0) {
            delete [] text;
            reader.close();
            cout << "Load Failed" << endl;
            return; 
        }
        delete [] text;
        
        vector<float> tempVec(dim, 0);
        for (int j = 0; j < dim; ++j) {
            if (reader.read((char *)(&tempVec[j]), sizeof(float)).gcount() == 0) {
                cout << "Load Failed" << endl;
                reader.close();
                return; 
            }
        }
        repoVec.push_back(tempVec);
    }

    // 加载请求向量
    for (int i = 0; i < querySize; ++i) {
        int length = 0;
        reader.read((char *)&length, 1 * sizeof(int));
        char *text = new char[length + 1];
        memset(text, 0, (length + 1) * sizeof(char));
        reader.read(text, length * sizeof(char));
        delete [] text;

        vector<float> tempVec(dim);
        for (int j = 0; j < dim; ++j) {
            if (reader.read((char *)(&tempVec[j]), sizeof(float)).gcount() == 0) {
                cout << "Load Failed" << endl;
                reader.close();
                return; 
            }
        }
        queryVec.push_back(tempVec);
    }
    reader.close();

    cout << "The Repos Size: " << repoVec.size() << endl;
    cout << "The Query Size: " << queryVec.size() << endl;
}


double RecallStatistics(TopKReCall& hnswReCall, TopKReCall& bruteReCall) {
    long long correct = 0, total = 0;
    if (hnswReCall.size() != bruteReCall.size()) { cout << "Failed to Check Data" << endl; return 0.0; }

    for (int i = 0; i < hnswReCall.size(); ++i) {
        set<idx_t> tab;
        for (auto &item : bruteReCall[i]) tab.insert(item.second);

        for (auto &item : hnswReCall[i]) {
            if (tab.find(item.second) != tab.end()) correct += 1;
        }
        total += tab.size();
        if (bruteReCall[i].size() != hnswReCall[i].size()) 
            cout << "Bad Data Size: " << bruteReCall[i].size() << "|" << hnswReCall[i].size() << endl;
    }
    
    if (total == 0) return 0.0;
    return 1.0f * correct / total;
}


float IP(vector<float>& a, vector<float>& b, int size) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void CompareIP(TopKReCall& ReCall, QueryVec query, QueryVec repos) {
    for (int i = 0; i < ReCall.size(); ++i) {
        for (int j = 0; j < ReCall[i].size(); ++j) {
            float res = IP(query[i], repos[ReCall[i][j].second], 256);
            cout << "ReCall: " << ReCall[i][j].first << " IP: " << res << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    cout << "HNSW Performance Test" << endl;

    if (argc == 2 && !strcmp(argv[1], "--help"))
        cout << "Usage: ./TestBase [DataFile] [repoSize] [querySize] [topK] [dims]" << endl;
    
    string resourceFile(argv[1]);
    int repoSize = atoi(argv[2]);
    int querySize = atoi(argv[3]);
    size_t topK = atol(argv[4]);
    int dims = atoi(argv[5]);

    cout << "Resource File: " << resourceFile << endl;
    cout << "Repos Size: " << repoSize << endl;
    cout << "Query Size: " << querySize << endl;
    cout << "Curr topK: " << topK << endl;
    cout << "Curr dims: " << dims << endl;

    hnswlib::InnerProductSpace space(dims);


    vector<vector<float>> repoDataVec, queryDataVec;
    cout << "Start to Load Resource" << endl;
    TimeType startTime = GetWatchTimer();
    LoadRepoData(resourceFile, dims, repoSize, querySize, repoDataVec, queryDataVec);
    cout << "Load Data using: " << GetWatchTimer(startTime) << endl;


    cout << "Start to Build Brute Index" << endl;
    startTime = GetWatchTimer();
    Index* bruteIndex = BuildBruteIndex(repoDataVec, space);
    cout << "Build Brute Index Using: " << GetWatchTimer(startTime) << endl;

    cout << "Start to Build HNSW Index" << endl;
    startTime = GetWatchTimer();
    Index* hnswIndex = BuildHnswIndex(repoDataVec, space);
    cout << "Build HNSW Index Using: " << GetWatchTimer(startTime) << endl;

    cout << "Start to Build Fiass Index" << endl;
    startTime = GetWatchTimer();
    FaissIndex* faissIndex = BuildFaissIndex(repoDataVec);
    cout << "Build Fiass Index Using: " << GetWatchTimer(startTime) << endl;


    TopKReCall bruteSearchAnswer;
    cout << "Start Brute Search" << endl;
    startTime = GetWatchTimer();
    SearchKCloserFirst(bruteIndex, topK, queryDataVec, bruteSearchAnswer);
    cout << "Brute Search Using: " << GetWatchTimer(startTime) << endl;
    delete bruteIndex;

    TopKReCall hnswSearchAnswer;
    cout << "Start HNSW Search" << endl;
    startTime = GetWatchTimer();
    SearchKCloserFirst(hnswIndex, topK, queryDataVec, hnswSearchAnswer);
    cout << "HNSW Search Using: " << GetWatchTimer(startTime) << endl;
    hnswIndex->saveIndex("./index.bin");
    delete hnswIndex;

    TopKReCall faissSearchAnswer;
    cout << "Start Faiss Search" << endl;
    startTime = GetWatchTimer();
    SearchKCloserFirst(faissIndex, topK, queryDataVec, faissSearchAnswer);
    faiss::write_index(faissIndex, "./faiss.index");
    cout << "Faiss Search Using: " << GetWatchTimer(startTime) << endl;

    // CompareIP(bruteSearchAnswer, queryDataVec, repoDataVec);

    cout << "The ReCall Rate of HNSW is: " << RecallStatistics(hnswSearchAnswer, bruteSearchAnswer) << endl;
    cout << "The ReCall Rate of Faiss is: " << RecallStatistics(faissSearchAnswer, bruteSearchAnswer) << endl;
    return 0;
}
