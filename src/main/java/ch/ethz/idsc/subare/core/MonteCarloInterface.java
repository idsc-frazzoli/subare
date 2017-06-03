// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.tensor.Tensor;

public interface MonteCarloInterface extends DiscreteModel, SampleModel, TerminalInterface {
  // ---
  Tensor startStates();
}
