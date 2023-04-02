//
// Created by 付聪 on 2017/6/26.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>


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

int main(int argc, char** argv){
  std::ios_base::sync_with_stdio(false);
  if(argc!=9){std::cout<< argv[0] <<" data_file init_graph save_graph K L iter S R"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned points_num, dim;
  dim = 100;
  // load_data(argv[1], data_load, points_num, dim);
  points_num = ReadBin(argv[1], dim, data_load);
  char* init_graph_filename = argv[2];
  char* graph_filename = argv[3];
  unsigned K = (unsigned)atoi(argv[4]);
  unsigned L = (unsigned)atoi(argv[5]);
  unsigned iter = (unsigned)atoi(argv[6]);
  unsigned S = (unsigned)atoi(argv[7]);
  unsigned R = (unsigned)atoi(argv[8]);
  data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));

  // index.Load(init_graph_filename);
  index.LoadNNFormat(init_graph_filename, K);
  std::cout << "NN-Descent refine Load graph finish." << std::endl;
  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);

  auto s = std::chrono::high_resolution_clock::now();

  index.RefineGraph(data_load, paras);

  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";

  index.Save(graph_filename);
  std::cout << "Save refine nndescent finish" << std::endl;
  return 0;
}
