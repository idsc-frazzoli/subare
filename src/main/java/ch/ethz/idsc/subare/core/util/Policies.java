// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.tensor.Tensor;

public enum Policies {
  ;
  // ---
  public static void print(PolicyInterface policyInterface, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policyInterface;
    eGreedyPolicy.print(states);
  }

  public static Tensor flatten(PolicyInterface policyInterface, Tensor states) {
    EGreedyPolicy eGreedyPolicy = (EGreedyPolicy) policyInterface;
    return eGreedyPolicy.flatten(states);
  }
  // TODO implement equals check
}
