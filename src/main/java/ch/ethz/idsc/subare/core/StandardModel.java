// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public interface StandardModel extends //
    MoveInterface, RewardInterface {
  // ---
  Tensor actions();

  // TODO not sure if this is good name
  Scalar qsa(Tensor state, Tensor action, Tensor gvalues);
}
