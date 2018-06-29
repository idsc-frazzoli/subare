//code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.tensor.Tensor;

public interface AnalysisAlgorithm {
  public String getName();

  public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa);
}
