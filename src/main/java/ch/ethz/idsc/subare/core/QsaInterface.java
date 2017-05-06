// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

@Deprecated
interface QsaInterface {
  Scalar qsa(Tensor state, Tensor action, Tensor values);
}
