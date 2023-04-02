//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_pq.h>
#include <efanna2e/util.h>
#include <omp.h>
#include <mimalloc-2.1/mimalloc.h>
//#include <chrono>


void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
uint32_t ReadBin(const std::string &file_path,
            //  std::vector<std::vector<float>> &data
            uint32_t& dim,
            float*& data
             ) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  // int cols = (dim + 7)/8*8;
  ifs.read((char *)&N, sizeof(uint32_t));
  // data = (float*)memalign(KGRAPH_MATRIX_ALIGN, N * cols * sizeof(float));
  data = new float[N * dim];
  std::cout << "# of points: " << N << std::endl;

  const int num_dimensions = 100;

  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    // data.push_back(buff);
    memcpy(data + counter * dim, buff.data(), num_dimensions * sizeof(float));
    counter++;
  }
  
  ifs.close();
  std::cout << "Finish Reading Data" << std::endl;
  return N;
}

void save_result(char* filename, std::vector<std::vector<unsigned> > &results){
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned) results[i].size();
    // out.write((char *) &GK, sizeof(unsigned));
    out.write((char *) results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}
void save_result(char* filename, faiss::idx_t *& results, unsigned points_num, unsigned K){
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  unsigned GK = (unsigned) K;
  for (unsigned i = 0; i < points_num; i++) {
    // out.write((char *) &GK, sizeof(unsigned));
    out.write((char *) results + i * K, GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv){
  std::ios_base::sync_with_stdio(false);
  if(argc!=6){std::cout<< argc << argv[0] <<" data_file IndexKey SearchKey K saving_graph"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned points_num, dim;
  dim = 100;
  // load_data(argv[1], data_load, points_num, dim);
  points_num = ReadBin(argv[1], dim, data_load);

  std::string pq_index_key(argv[2]);
  std::string pq_search_key(argv[3]);
  unsigned K = (unsigned)atoi(argv[4]);

  data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
  efanna2e::IndexPQ index(dim, points_num, efanna2e::L2, nullptr);

  // omp_set_num_threads(32);

  auto s = std::chrono::high_resolution_clock::now();

  efanna2e::Parameters paras;
  paras.Set<std::string>("pq_index_key", pq_index_key);

  index.Build(points_num, data_load, paras);

  paras.Set<std::string>("pq_search_key", pq_search_key);

  std::vector<std::vector<unsigned> > res(points_num);
  std::cout << "search begin" << std::endl;
  // int cnt = 0;
  // int k = 100;
  // faiss::idx_t *Ids = new faiss::idx_t[points_num * k];
  // float *Dists = new float[points_num * k];
  // index.index->search(points_num, data_load, k, Dists, Ids);
#pragma omp parallel for
  for(unsigned i=0; i<points_num; i++){
    std::vector<unsigned> &tmp = res[i];
    tmp.resize(K);
    index.Search(data_load + i * dim, data_load, K, paras, tmp.data());
    // cnt++;
    // if(cnt % 100000 == 0){
    //   std::cout<<cnt/100000<<" / "<<points_num/100000<<" complete\n";
    // }
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";


  save_result(argv[5], res);
  // save_result(argv[5], Ids, points_num, k);

  return 0;
}
