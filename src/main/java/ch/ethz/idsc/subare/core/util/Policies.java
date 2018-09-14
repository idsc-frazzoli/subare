// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public enum Policies {
  ;
  public static void print(Policy policy, Tensor states) {
    PolicyBase policyBase = (PolicyBase) policy;
    System.out.println("best actions:");
    for (Tensor state : states)
      System.out.println(state + " -> " + policyBase.getBestActions(state));
  }

  /** useful for export to Mathematica
   * 
   * @param states
   * @return list of state action pairs that are optimal with respect to ... */
  public static Tensor flatten(PolicyBase policyBase, Tensor states) {
    Tensor result = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : policyBase.getBestActions(state))
        result.append(StateAction.key(state, action));
    return result;
  }

  /** function builds a {@link DiscreteQsa} where the state-action values are
   * the evaluation of the bi-function {@link Policy#probability(Tensor, Tensor)}.
   * 
   * @param discreteModel
   * @param policy
   * @return */
  public static DiscreteQsa toQsa(DiscreteModel discreteModel, Policy policy) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    return qsa.create(qsa.keys().stream() //
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
