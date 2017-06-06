// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface MonteCarloInterface extends DiscreteModel, SampleModel, TerminalInterface {
  /** @return states that are candidates to start an episode */
  Tensor startStates();
}
