// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Random;
import java.util.function.Supplier;

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
 * box on p.138
 * 
 * 2)
 * n-step Sarsa for estimating Q(s,a)
 * 
 * box on p.157 */
public class OriginalSarsa extends Sarsa {
  private static final Random RANDOM = new Random();
  // ---

  /** @param discreteModel
   * @param qsa
   * @param learningRate */
  public OriginalSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  @Override
  public void supplyPolicy(Supplier<Policy> supplier) {
    policy = supplier.get();
  }

  @Override
  protected Scalar evaluate(Tensor state) {
    return crossEvaluate(state, qsa);
  }

  @Override
  protected Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    PolicyWrap policyWrap = new PolicyWrap(policy, RANDOM);
    Tensor action = policyWrap.next(state, discreteModel.actions(state));
    return Qsa2.value(state, action);
  }
}
