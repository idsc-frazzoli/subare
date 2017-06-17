// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

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
  // not used yet
  public static Scalar p_distance(DiscreteModel discreteModel, Policy pi1, Policy pi2) {
    return DiscreteValueFunctions.distance( //
        toQsa(discreteModel, pi1), //
        toQsa(discreteModel, pi2));
  }

  // not used yet
  public static Scalar expectedQvalue(DiscreteModel discreteModel, QsaInterface qsa, Policy policy) {
    Scalar sum = RealScalar.ZERO;
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        sum = sum.add(policy.probability(state, action).multiply(qsa.value(state, action)));
    return sum;
  }
}
