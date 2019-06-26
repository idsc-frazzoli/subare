// code by fluric
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.core.util.ExplorationRate;

@FunctionalInterface
public interface ExplorationRateSupplier {
  ExplorationRate explorationRate();
}
