// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;

public class Average {
  private Scalar value = null;
  private int count = 0;

  public void track(Scalar scalar) {
    if (count == 0) {
      value = scalar;
    } else {
      Scalar weight = RealScalar.of(count + 1).invert();
      // System.out.println(value);
      // System.out.println(weight);
      // System.out.println(scalar);
      value = value.multiply(RealScalar.ONE.subtract(weight)) //
          .add(scalar.multiply(weight));
    }
    ++count;
  }

  public Scalar get() {
    return value;
  }

  @Override
  public String toString() {
    return value.toString();
  }
}
