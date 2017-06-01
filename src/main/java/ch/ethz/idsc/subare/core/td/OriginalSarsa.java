// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** Sarsa: An on-policy TD control algorithm
 * 
 * eq (6.7)
 * 
 * box on p.138
 * 
 * the Sarsa algorithm was introduced by Rummery and Niranjan 1994 */
public class OriginalSarsa extends Sarsa {
  final PolicyWrap policyWrap;

  public OriginalSarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, //
      QsaInterface qsa, Scalar alpha) {
    super(episodeSupplier, policyInterface, discreteModel, qsa, alpha);
    policyWrap = new PolicyWrap(policyInterface);
  }

  @Override
  protected Scalar evaluate(Tensor state1) {
    Tensor action1 = policyWrap.next(state1, discreteModel.actions(state1));
    return qsa.value(state1, action1);
  }
}
