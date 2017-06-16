// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.TensorRuntimeException;
import ch.ethz.idsc.tensor.red.ArgMax;

public enum Policies {
  ;
  public static void print(Policy policy, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policy;
    eGreedyPolicy.print(states);
  }

  // function only used once
  public static Tensor flatten(Policy policy, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policy;
    return eGreedyPolicy.flatten(states);
  }

  /** function builds a {@link DiscreteQsa} where the state-action values are
   * the evaluation of the bi-function {@link Policy#probability(Tensor, Tensor)}.
   * 
   * @param discreteModel
   * @param policy
   * @return */
  public static DiscreteQsa toQsa(DiscreteModel discreteModel, Policy policy) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    return qsa.create(qsa.index.keys().flatten(0) //
        .map(sap -> policy.probability(sap.get(0), sap.get(1))));
  }

  /** @param discreteModel
   * @param pi1
   * @param pi2
   * @return true, if pi1 and pi2 are equal for all state-action pairs */
  public static boolean equals(DiscreteModel discreteModel, Policy pi1, Policy pi2) {
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Scalar value1 = pi1.probability(state, action);
        Scalar value2 = pi2.probability(state, action);
        if (!value1.equals(value2))
          return false;
      }
    return true;
  }

  /** @param discreteModel
   * @param pi1
   * @param pi2
   * @return distance between probabilities */
  public static Scalar p_distance(DiscreteModel discreteModel, Policy pi1, Policy pi2) {
    return TensorValuesUtils.distance( //
        toQsa(discreteModel, pi1), //
        toQsa(discreteModel, pi2));
  }

  /** @param discreteModel
   * @param ref ground truth
   * @param policy to compared against ref
   * @return non-negative expected loss */
  @Deprecated
  public static Scalar expectedLossOld(DiscreteModel discreteModel, QsaInterface ref, Policy policy) {
    Policy target = GreedyPolicy.bestEquiprobable(discreteModel, ref);
    Scalar max = expectedQvalue(discreteModel, ref, target);
    Scalar cmp = expectedQvalue(discreteModel, ref, policy);
    Scalar difference = max.subtract(cmp);
    if (Scalars.lessThan(difference, RealScalar.ZERO))
      throw TensorRuntimeException.of(max, cmp); // ref was not optimal
    return difference;
  }

  @Deprecated
  public static Scalar expectedLossOld(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return expectedLossOld(discreteModel, ref, GreedyPolicy.bestEquiprobable(discreteModel, qsa));
  }

  /** @param discreteModel
   * @param ref
   * @param qsa
   * @return */
  public static Scalar expectedLoss(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    Scalar sum = RealScalar.ZERO;
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      int ind_max = ArgMax.of(Tensor.of(actions.flatten(0).map(action -> ref.value(state, action))));
      Scalar max = ref.value(state, actions.get(ind_max));
      int ind_cmp = ArgMax.of(Tensor.of(actions.flatten(0).map(action -> qsa.value(state, action))));
      Scalar cmp = ref.value(state, actions.get(ind_cmp));
      if (Scalars.lessThan(max, cmp))
        throw TensorRuntimeException.of(cmp, max);
      sum = sum.add(max.subtract(cmp));
    }
    return sum;
  }

  public static Scalar expectedQvalue(DiscreteModel discreteModel, QsaInterface qsa, Policy policy) {
    Scalar sum = RealScalar.ZERO;
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        sum = sum.add(policy.probability(state, action).multiply(qsa.value(state, action)));
    return sum;
  }
}
