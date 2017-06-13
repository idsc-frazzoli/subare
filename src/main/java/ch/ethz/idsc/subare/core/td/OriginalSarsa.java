// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Tensor;

/** the Sarsa algorithm was introduced by Rummery and Niranjan (1994) as
 * "Modified Connectionist Q-learning"
 * 
 * 1)
 * Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.7)
 * 
 * box on p.138
 * 
 * 2)
 * n-step Sarsa for estimating Q(s,a)
 * 
 * box on p.157 */
public class OriginalSarsa extends ActionSarsa {
  private static final Random random = new Random();
  // ---

  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate
   * @param policyInterface */
  public OriginalSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override
  Tensor selectAction(Tensor state) {
    PolicyWrap policyWrap = new PolicyWrap(policyInterface, random);
    return policyWrap.next(state, discreteModel.actions(state));
  }
}
