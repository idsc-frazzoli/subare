// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.NumberQ;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Clips;
import ch.ethz.idsc.tensor.sca.ScalarUnaryOperator;

/** clips scalars to interval [0, 1] excluding scalars that do not satisfy {@link NumberQ}
 * such as {@link DoubleScalar#POSITIVE_INFINITY} and {@link DoubleScalar#INDETERMINATE} */
/* package */ enum UnitClip implements ScalarUnaryOperator {
  FUNCTION;
  // ---
  @Override
  public Scalar apply(Scalar scalar) {
    return NumberQ.of(scalar) //
        ? Clips.unit().apply(scalar)
        : scalar;
  }

  @SuppressWarnings("unchecked")
  public static <T extends Tensor> T of(T tensor) {
    return (T) tensor.map(FUNCTION);
  }
}
