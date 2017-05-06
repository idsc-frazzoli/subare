// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface EpisodeInterface {
  /** @return current state */
  Tensor state();

  /** @return */
  StepInterface step();

  boolean hasNext();
}
