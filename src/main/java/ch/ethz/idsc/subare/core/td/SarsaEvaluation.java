// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/* package */ interface SarsaEvaluation {
  Scalar evaluate(Tensor state, PolicyBase policy);

  /** The action probabilities are chosen according to policy1 and then added up by the qsa of policy2
   * @param state
   * @param policy1
   * @param policy2
   * @return qsa value */
  Scalar crossEvaluate(Tensor state, PolicyBase policy1, PolicyBase policy2);
}
