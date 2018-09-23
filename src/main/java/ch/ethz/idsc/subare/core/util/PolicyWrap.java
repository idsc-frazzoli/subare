// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionModel;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.EmpiricalDistribution;
import ch.ethz.idsc.tensor.pdf.RandomVariate;

/** class picks action based on distribution defined by given {@link Policy} */
public class PolicyWrap {
  private final Policy policy;

  public PolicyWrap(Policy policy) {
    this.policy = policy;
  }

  /** @param state
   * @param actions non-empty subset of all possible actions from given state
   * @return */
  public Tensor next(Tensor state, Tensor actions) {
    Tensor pdf = Tensor.of(actions.stream().map(action -> policy.probability(state, action)));
    Distribution distribution = EmpiricalDistribution.fromUnscaledPDF(pdf);
    return actions.get(RandomVariate.of(distribution).number().intValue());
  }

  /** @param state
   * @param stateActionModel
   * @return */
  public Tensor next(Tensor state, StateActionModel stateActionModel) {
    Distribution distribution = policy.getDistribution(state);
    return stateActionModel.actions(state).get(RandomVariate.of(distribution).number().intValue());
  }
}
