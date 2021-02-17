// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.api.TensorScalarFunction;
import ch.ethz.idsc.tensor.nrm.Vector1Norm;
import ch.ethz.idsc.tensor.sca.InvertUnlessZero;
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

  /** @param tvi1
   * @param tvi2
   * @param norm for vectors
   * @return */
  public static Scalar distance(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2, TensorScalarFunction norm) {
    return norm.apply(_difference(tvi1, tvi2));
  }

  public static Scalar distance(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return distance(tvi1, tvi2, Vector1Norm::of);
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

  /** @param qsa1
   * @param qsa2
   * @param sac1
   * @param sac2
   * @return the weighted average of the qsa values according to the number of visits occurred in the different {@link LearningRate}'s.
   * For each element of the qsa: qsa(e) = (qsa1(e)*lr1_visits(e) + qsa2(e)*lr2_visits(e))/(lr1_visits(e)+lr2_visits(e)) */
  public static DiscreteQsa weightedAverage(DiscreteQsa qsa1, DiscreteQsa qsa2, //
      StateActionCounter sac1, StateActionCounter sac2) {
    Tensor visits1 = Tensor.of(qsa1.keys().stream().map(sac1::stateActionCount));
    Tensor visits2 = Tensor.of(qsa2.keys().stream().map(sac2::stateActionCount));
    Tensor inverse = visits1.add(visits2).map(InvertUnlessZero.FUNCTION);
    return qsa1.create(qsa1.values().pmul(visits1).add(qsa2.values().pmul(visits2)).pmul(inverse).stream());
  }

  /**************************************************/
  // helper function
  private static boolean _isCompatible(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return tvi1.keys().equals(tvi2.keys());
  }

  // helper function
  private static Tensor _difference(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    if (_isCompatible(tvi1, tvi2))
      return tvi1.values().subtract(tvi2.values());
    throw new IllegalArgumentException();
  }
}
