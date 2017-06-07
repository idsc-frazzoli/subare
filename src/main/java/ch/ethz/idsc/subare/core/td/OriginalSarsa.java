// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Random;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** the Sarsa algorithm was introduced by Rummery and Niranjan 1994
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
  private final Random random = new Random();
  private final PolicyInterface policyInterface;

  /** @param discreteModel
   * @param qsa
   * @param alpha learning rate
   * @param policyInterface */
  public OriginalSarsa( //
      DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha, //
      PolicyInterface policyInterface) {
    super(discreteModel, qsa, alpha);
    this.policyInterface = policyInterface;
  }

  @Override
  protected Scalar evaluate(Tensor state1) {
    PolicyWrap policyWrap = new PolicyWrap(policyInterface, random);
    Tensor action1 = policyWrap.next(state1, discreteModel.actions(state1));
    return qsa.value(state1, action1);
  }
}
