#include "bits/stdc++.h"
#include "UtilTools.hpp"

#include "faiss/IndexIVF.h"
#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/impl/io.h"
#include "faiss/index_io.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/impl/AuxIndexStructures.h"

using namespace std;
using namespace utiltools;

using QueryVec = vector<vector<float>>;
using TopKReCall = vector<vector<pair<float, size_t>>>;
using FaissIndexPtr = shared_ptr<faiss::IndexIVFFlat>;

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


FaissIndexPtr TransIndex2IndexPtr(faiss::IndexIVFFlat &index) {
    faiss::VectorIOWriter indexWriter;
    faiss::write_index(&index, &indexWriter);

    faiss::VectorIOReader indexReader;
    indexReader.data = indexWriter.data;
    faiss::IndexIVFFlat* newIndexPtr = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(&indexReader));

    FaissIndexPtr indexSharedPtr;
    indexSharedPtr.reset(newIndexPtr);
    return indexSharedPtr;
}


FaissIndexPtr BuildFaissIndex(vector<vector<float>> &dataVec) {
    if (dataVec.size() == 0) { cout << "Empty Data" << endl; return NULL; }
    int lens = dataVec.size();
    int dims = dataVec[0].size();

    long *index_id = new long[lens];
    float *feature = new float[lens * dims];
    memset(index_id, 0, lens * sizeof(long));
    memset(feature, 0, lens * dims * sizeof(float));

    for (int i = 0; i < lens; ++i) {
        index_id[i] = i;
        copy(dataVec[i].begin(), dataVec[i].end(), &feature[i * dims]);
    }

    // 参数的设置参考检索问答
    faiss::IndexFlatL2 quantizer(dims);
    const long nList = lens > 10000 ? 4 * sqrt(lens) : 1;
    faiss::IndexIVFFlat index(&quantizer, dims, nList, faiss::METRIC_INNER_PRODUCT);
    index.train(lens, feature);
    index.add_with_ids(lens, feature, index_id);

    delete [] index_id;
    delete [] feature;
    return TransIndex2IndexPtr(index);
}


void SearchKCloserFirst(FaissIndexPtr faissIndex, const size_t topK, QueryVec& dataVec, TopKReCall& reCallVec) {
    if (NULL == faissIndex) { cout << "NULL of Index Error" << endl; return; }
    for (int i = 0; i < dataVec.size(); ++i) { 
        float score[topK];
        long  index[topK];
        faissIndex->search(1, dataVec[i].data(), topK, score, index);

        vector<pair<float, size_t>> resp(topK);
        for (int i = 0; i < topK; ++i) resp[i] = make_pair(score[i], index[i]);
        reCallVec.push_back(resp);
    }
}

int main (int argc, char* argv[]) {
    cout << "Faiss Performance Test" << endl;

    if (argc == 2 && !strcmp(argv[1], "--help"))
        cout << "Usage: ./FaissTest [DataFile] [repoSize] [querySize] [topK] [dims]" << endl;
    
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

    vector<vector<float>> repoDataVec, queryDataVec;
    cout << "Start to Load Resource" << endl;
    TimeType startTime = GetWatchTimer();
    LoadRepoData(resourceFile, dims, repoSize, querySize, repoDataVec, queryDataVec);
    cout << "Load Data using: " << GetWatchTimer(startTime) << endl;


    FaissIndexPtr faissIndex = BuildFaissIndex(repoDataVec);

    TopKReCall faissReCall;
    SearchKCloserFirst(faissIndex, topK, queryDataVec, faissReCall);

    return 0;
}