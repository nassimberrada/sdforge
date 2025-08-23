float sgn(float x) 
{
	return (x<0.0)?-1.0:1.0;
}
float sigmoid(float x) 
{
    return 1.0 / (1.0 + exp(-x));
}
float tanh(float x) 
{
  float ex = exp(x);
  float enx = 1.0 / ex;
  return (ex - enx) / (ex + enx);
}