#include <sstream>
#include <vector>
#include <string>

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> tokens;
  std::istringstream iss(s);
  std::string token;
  while (iss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::string join(
    const std::vector<std::string>& elements,
    const std::string& delimiter
) {
  std::string result;
  for (size_t i = 0; i < elements.size(); ++i) {
    result += elements[i];
    if (i != elements.size() - 1) {
      result += delimiter;
    }
  }
  return result;
}
