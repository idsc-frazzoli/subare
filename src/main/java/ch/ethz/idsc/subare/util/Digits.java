// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Round;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;

// TODO obsolete, once update of tensor lib
@Deprecated
public enum Digits implements ScalarUnaryOperator {
  _1("0.1"), //
  _2("0.01"), //
  _3("0.001"), //
  _4("0.0001"), //
  ;
  private final ScalarUnaryOperator round;

  private Digits(String digits) {
    round = Round.toMultipleOf(DecimalScalar.of(digits));
  }

  @Override
  public Scalar apply(Scalar digits) {
    return round.apply(digits);
  }
}
