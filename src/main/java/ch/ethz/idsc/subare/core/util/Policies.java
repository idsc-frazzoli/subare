// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public enum Policies {
  ;
  // ---
  public static void print(PolicyInterface policyInterface, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policyInterface;
    eGreedyPolicy.print(states);
  }

  // function only used once
  public static Tensor flatten(PolicyInterface policyInterface, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policyInterface;
    return eGreedyPolicy.flatten(states);
  }

  /** function builds a {@link DiscreteQsa} where the state-action values are
   * the evaluation of the bi-function {@link PolicyInterface#policy(Tensor, Tensor)}.
   * 
   * @param discreteModel
   * @param policyInterface
   * @return */
  public static DiscreteQsa toQsa(DiscreteModel discreteModel, PolicyInterface policyInterface) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    return qsa.create(qsa.index.keys().flatten(0).map(sap -> policyInterface.policy(sap.get(0), sap.get(1))));
  }

  /** @param discreteModel
   * @param pi1
   * @param pi2
   * @return true, if pi1 and pi2 are equal for all state-action pairs */
  public static boolean equals(DiscreteModel discreteModel, PolicyInterface pi1, PolicyInterface pi2) {
    // TODO implement using streams
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Scalar value1 = pi1.policy(state, action);
        Scalar value2 = pi2.policy(state, action);
        if (!value1.equals(value2))
          return false;
      }
    return true;
  }
}
