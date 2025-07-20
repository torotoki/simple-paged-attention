#include <stdexcept>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <chrono>
#include <cassert>

#include <cuda_runtime.h>

using namespace std;

class Stopwatch {
  public:
    Stopwatch(string mode = "chrono") : mode(mode) {
      set<string> options = {"chrono", "cuda"};
      assert(options.count(mode));
      if (mode == "cuda") {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);
      }
    }

    ~Stopwatch() {
      if (mode == "cuda") {
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
      }
    }

    void start() {
      assert(intervals.size() == 0 && !finished);
      if (mode == "chrono")
        intervals.push_back(chrono::system_clock::now());
      else
        cudaEventRecord(cuda_start);
    }

    void pause() {
      if (mode == "chrono") {
        assert(intervals.size() % 2 == 1 && !finished);
        intervals.push_back(chrono::system_clock::now());
      } else {
        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
      }
    }

    void resume() {
      if (mode == "chrono") {
        assert(intervals.size() % 2 == 0 && !finished);
        intervals.push_back(chrono::system_clock::now());
      } else
        throw std::invalid_argument(
            "CUDA timer does not support resume() yet"
        );
    }

    void lap() {
      assert(intervals.size() % 2 == 1 && !finished);
      chrono::system_clock::time_point now = chrono::system_clock::now();
      intervals.push_back(now);
      intervals.push_back(now);
    }

    void stop() {
      pause();
      finished = true;
    }

    double get_elapsed_time_msec() {
      assert(finished);
      if (mode == "chrono")
        return chrono::duration<double, std::milli>
          (intervals.back() - intervals.front()).count();
      else {
        float msec = 0.0f;
        cudaEventElapsedTime(&msec, cuda_start, cuda_stop);
        return (double)msec;
      }
    }

    void pprint(string tag = "") {
      // Incomplete intervals are not supported yet.
      assert(intervals.size() % 2 == 0);
      double elapsed_time_msec = get_elapsed_time_msec();
      if (tag != "") {
        cout << tag << ": ";
      }
      cout << "Total Time: " << elapsed_time_msec << " msec" << endl;
      
      if (intervals.size() == 2)
        return;

      vector<double> time_elapsed_msecs;
      for (int i = 0; i < intervals.size(); i += 2) {
        chrono::system_clock::time_point start, end;
        double msec;
        start = intervals[i];
        end = intervals[i + 1];
        msec = chrono::duration<double, std::milli>(end - start).count();
        time_elapsed_msecs.push_back(msec);
      }

      for (int i = 0; i < time_elapsed_msecs.size(); ++i) {
        cout << "Time("<< i <<"): " << time_elapsed_msecs[i] << " msec" << endl;
      }
    }

  private:
    vector<chrono::system_clock::time_point> intervals;
    bool finished = false;
    string mode;
    cudaEvent_t cuda_start, cuda_stop;
};

