find . -name "*.py" -exec black {} +
find . -name "*.cpp" -exec clang-format -i {} +
find . -name "*.hpp" -exec clang-format -i {} +

