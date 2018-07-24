// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;

public interface MonteCarloTrial extends DiscreteQsaSupplier {
  void executeBatch();

  void digest(StepInterface stepInterface);

  QsaInterface qsaInterface();
}
