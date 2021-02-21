// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.subare.util.FairArg;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.EmpiricalDistribution;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions */
/* package */ class UcbPolicy extends PolicyBase {
  /* package */ UcbPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
  }

  /* package */ UcbPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
  }

  @Override // from PolicyBase
  public Tensor getBestActions(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().parallel() //
        .map(action -> UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, action), sac, discreteModel)));
    FairArg fairArg = FairArg.max(va);
    return Tensor.of(fairArg.options().stream().map(actions::get));
  }

  @Override // from Policy
  public Distribution getDistribution(Tensor state) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> index.containsKey(action) ? RationalScalar.of(1, optimalCount) : RealScalar.ZERO));
    return EmpiricalDistribution.fromUnscaledPDF(pdf);
  }

  @Override // from Policy
  public Scalar probability(Tensor state, Tensor action) {
    Tensor actions = getBestActions(state);
    return actions.stream().anyMatch(action::equals) // computational complexity is O(n)
        ? RationalScalar.of(1, actions.length())
        : RealScalar.ZERO;
  }

  @Override // from PolicyBase
  public UcbPolicy copy() {
    return new UcbPolicy(discreteModel, qsa, sac);
  }
}
