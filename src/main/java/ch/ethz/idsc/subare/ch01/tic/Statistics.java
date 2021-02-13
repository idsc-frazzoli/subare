// code by jph
// http://stackoverflow.com/questions/7988486/how-do-you-calculate-the-variance-median-and-standard-deviation-in-c-or-java
package ch.ethz.idsc.subare.ch01.tic;

import java.util.Arrays;

import ch.ethz.idsc.tensor.ext.Integers;

/* package */ class Statistics {
  double[] data;
  int size;

  Statistics(double[] data) {
    this.data = data;
    size = data.length;
  }

  double getMean() {
    double sum = 0.0;
    for (double a : data)
      sum += a;
    return sum / size;
  }

  double getVariance() {
    double mean = getMean();
    double temp = 0;
    for (double a : data)
      temp += (a - mean) * (a - mean);
    return temp / size;
  }

  double getStdDev() {
    return Math.sqrt(getVariance());
  }

  public double median() {
    Arrays.sort(data);
    int mid = data.length / 2;
    return Integers.isEven(data.length) //
        ? (data[mid - 1] + data[mid]) / 2.0
        : data[mid];
  }
}
