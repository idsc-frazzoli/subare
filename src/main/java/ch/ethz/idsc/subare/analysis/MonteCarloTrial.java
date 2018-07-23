// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;

public interface MonteCarloTrial extends DiscreteQsaSupplier {
  void executeBatch();
}
