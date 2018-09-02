// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.TrueOnlineInterface;

public interface MonteCarloTrial extends TrueOnlineInterface {
  void executeBatch();
}
