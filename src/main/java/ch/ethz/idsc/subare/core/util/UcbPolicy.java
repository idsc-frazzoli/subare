// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions */
// TODO fluric right now there is no reason to make the class public
// ... since construction happens exclusively via policytype
/* package */ class UcbPolicy extends PolicyBase {
  /* package */ UcbPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
  }

  /* package */ UcbPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
  }

  @Override
  public Tensor getBestActions(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().parallel(). //
        map(action -> UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, action), sac, discreteModel)));
    FairArgMax fairArgMax = FairArgMax.of(va);
    return Tensor.of(fairArgMax.options().stream().map(actions::get));
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Index index = Index.build(getBestActions(state));
    final int optimalCount = index.size();
    return index.containsKey(action) ? RationalScalar.of(1, optimalCount) : RealScalar.ZERO;
  }

  @Override
  public PolicyBase copyOf(PolicyBase policyBase) {
    return new UcbPolicy(policyBase.discreteModel, policyBase.qsa, policyBase.sac);
  }
}
