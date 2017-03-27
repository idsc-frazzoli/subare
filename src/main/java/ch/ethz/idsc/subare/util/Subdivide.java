package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public class Subdivide {
  public static Tensor range(Scalar beg, Scalar end, int length) {
    Scalar dif = end.subtract(beg).divide(RealScalar.of(length));
    Tensor sup = Tensors.empty();
    sup.append(beg);
    for (int c = 1; c < length - 1; ++c) {
      sup.append(beg.add(dif.multiply(RealScalar.of(c))));
    }
    sup.append(end);
    GlobalAssert.of(sup.length() == length);
    return sup;
  }
}
