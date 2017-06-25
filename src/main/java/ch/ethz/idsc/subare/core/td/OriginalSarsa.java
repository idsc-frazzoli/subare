// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.RealScalar;
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
  /** @param discreteModel
   * @param qsa
   * @param learningRate */
  public OriginalSarsa(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    super(discreteModel, qsa, learningRate);
  }

  // TODO can refactor between original sarsa and expected sarsa
  @Override
  protected Scalar evaluate(Tensor state) {
    return crossEvaluate(state, qsa);
  }

  @Override
  protected Scalar crossEvaluate(Tensor state, QsaInterface Qsa2) {
    Tensor actions = Tensor.of( //
        discreteModel.actions(state).flatten(0) //
            .filter(action -> learningRate.encountered(state, action)));
    if (actions.length() == 0)
      return RealScalar.ZERO;
    // ---
    Policy policy = EGreedyPolicy.bestEquiprobable(discreteModel, Qsa2, epsilon, state);
    Tensor action = new PolicyWrap(policy).next(state, actions);
    return Qsa2.value(state, action);
  }
}
