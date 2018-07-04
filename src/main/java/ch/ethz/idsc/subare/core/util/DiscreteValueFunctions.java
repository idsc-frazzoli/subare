// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.red.Norm;
import ch.ethz.idsc.tensor.sca.LogisticSigmoid;
import ch.ethz.idsc.tensor.sca.N;

public enum DiscreteValueFunctions {
  ;
  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T numeric(T tvi) {
    return (T) tvi.create(N.DOUBLE.of(tvi.values()).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T rescaled(T tvi) {
    return (T) tvi.create(Rescale.of(tvi.values()).stream());
  }

  public static Scalar distance(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return Norm._1.ofVector(_difference(tvi1, tvi2));
  }

  public static Scalar distanceSquared(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return Norm._2.ofVector(_difference(tvi1, tvi2));
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T average(T tvi1, T tvi2) {
    return (T) tvi1.create(tvi1.values().add(tvi2.values()).multiply(RationalScalar.HALF).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T logisticDifference(T tvi1, T tvi2) {
    return (T) tvi1.create(LogisticSigmoid.of(_difference(tvi1, tvi2)).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T logisticDifference(T tvi1, T tvi2, Scalar factor) {
    return (T) tvi1.create(LogisticSigmoid.of(_difference(tvi1, tvi2).multiply(factor)).stream());
  }

  /**************************************************/
  // helper function
  private static boolean _isCompatible(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return tvi1.keys().equals(tvi2.keys());
  }

  // helper function
  private static Tensor _difference(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    if (!_isCompatible(tvi1, tvi2))
      throw new RuntimeException();
    return tvi1.values().subtract(tvi2.values());
  }
}
