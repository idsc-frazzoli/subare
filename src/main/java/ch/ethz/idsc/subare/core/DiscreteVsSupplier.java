// code by jph
package ch.ethz.idsc.subare.core;

import ch.ethz.idsc.subare.core.util.DiscreteVs;

@FunctionalInterface
public interface DiscreteVsSupplier {
  DiscreteVs vs();
}
