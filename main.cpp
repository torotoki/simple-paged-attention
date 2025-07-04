#include <iostream>
#include <cassert>
#include <stdexcept>
#include <map>
#include <string>

#include "common/host_utils.h"
#include "common/string_utils.cpp"
#include "common/timer.cpp"
#include "cpu_attention.hpp"

using namespace std;


void runBenchmarkOneIter(
  string command,
  int context_size = 14,
  int d_model = 256,      // embedding dimention
  int d_k = 128,          // Q's, K's and V's hidden dimention
  bool verbose = true
) {
  /**
   * Input:
   *   - W_Q, W_K, W_V: d_model x d_k
   *     -> Q = X x W_Q : context_size x d_k
   *     -> Q x K^T : context_size x context_size
   *     -> (softmax(Q x K^T) / d_k) x V : context_size x d_k
   *   - words: vector<string>
   *     -> X: context_size x d_model
   */
  string input = "The fire burns twice as bright burns half as long .";
  vector<string> words = split(input);
  if (verbose)
    cout << "Input words: " << input << endl;

  // Generate embeddings
  map<string, vector<float>> embedding_table;
  Matrix<float> X(context_size, d_model);
  InputGenerator generator = InputGenerator();

  for (size_t i = 0; i < words.size(); ++i) {
    vector<float> embedding;
    if (embedding_table.find(words[i]) != embedding_table.end()) {
      embedding = embedding_table[words[i]];
    } else {
      embedding = generator.generateRandomVector<float>(d_model, -0.25f, 0.25f);
      embedding_table[words[i]] = embedding;
    }
    X.set_row(i, embedding);
  }

  Matrix W_Q = generator.generateConstantMatrix<float>(d_model, d_k);
  Matrix W_K = generator.generateConstantMatrix<float>(d_model, d_k);
  Matrix W_V = generator.generateConstantMatrix<float>(d_model, d_k);

  if (command == "cpu_attention") {
    Matrix<float> computed = compute_attention_on_cpu(context_size, d_model, d_k, W_Q, W_K, W_V, X);
    for (int i = 0; i < words.size(); ++i) {
      for (int j = 0; j < d_k; ++j) {
        cout << computed.at(i, j) << " ";
      }
      cout << endl;
    }
    cout << endl;
  } else if (command == "attention") {
    // Matrix<float> computed = launch_attention_kernel(context_size, d_model, d_k, W_Q, W_K, W_V, X);
  } else {
    throw std::runtime_error("Unknown command: " + command);
  }
}

int main(int argc, char* argv[]) {
  string command = "cpu_attention";
  if (argc >= 2) {
    command = argv[1];
  }
  
  cout << "Command: " << command << endl;
  runBenchmarkOneIter(command);
}
