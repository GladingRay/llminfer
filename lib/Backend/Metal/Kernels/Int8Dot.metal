inline int dotI8x4(char4 a, char4 b) {
  int4 products = int4(a) * int4(b);
  return products.x + products.y + products.z + products.w;
}
