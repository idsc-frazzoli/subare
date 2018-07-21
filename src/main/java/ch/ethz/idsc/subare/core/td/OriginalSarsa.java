// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** the Sarsa algorithm was introduced by Rummery and Niranjan (1994) as
 * "Modified Connectionist Q-learning"
 * 
 * 1)
 * Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.7)
 * 
 * box on p.130
 * 
 * 2)
 * n-step Sarsa for estimating Q(s,a)
 * 
 * box on p.157 */
/* package */ class OriginalSarsa extends AbstractSharedSarsa {
  OriginalSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override // from AbstractSharedSarsa
  Scalar crossEvaluate(Tensor state, Tensor actions, QsaInterface Qsa2, Policy policy) {
    Tensor action = new PolicyWrap(policy).next(state, actions);
    return Qsa2.value(state, action);
  }
}
