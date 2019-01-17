// code by jph
package ch.ethz.idsc.subare.demo.fish;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Exp;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;

enum DLogisticSigmoid implements ScalarUnaryOperator {
  FUNCTION;
  // ---
  @Override
  public Scalar apply(Scalar scalar) {
    Scalar exp = Exp.of(scalar); // Exp[x]
    Scalar den = RealScalar.ONE.add(exp); // 1+Exp[x]
    return exp.divide(den.multiply(den)); // Exp[x] / (1+Exp[x])^2
  }
}
