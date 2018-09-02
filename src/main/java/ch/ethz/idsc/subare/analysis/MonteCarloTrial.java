// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.TrueOnlineInterface;

// TODO clarify what is difference to function qsa() ?
public interface MonteCarloTrial extends TrueOnlineInterface {
  void executeBatch();
}
