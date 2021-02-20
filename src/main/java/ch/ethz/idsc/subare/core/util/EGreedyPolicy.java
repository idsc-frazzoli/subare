// code by jph, fluric
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
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.pdf.Distribution;
import ch.ethz.idsc.tensor.pdf.EmpiricalDistribution;

/** p.33 */
public class EGreedyPolicy extends PolicyBase {
  // LONGTERM make explorationRate final
  private ExplorationRate explorationRate;

  public EGreedyPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
    explorationRate = ConstantExplorationRate.of(0.1); // TODO magic const
  }

  public EGreedyPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
    explorationRate = ConstantExplorationRate.of(0.1); // TODO magic const
  }

  public void setExplorationRate(ExplorationRate explorationRate) {
    this.explorationRate = explorationRate;
  }

  @Override
  public Tensor getBestActions(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArg fairArgMax = FairArg.max(va);
    return Tensor.of(fairArgMax.options().stream().map(actions::get));
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    final int nonOptimalCount = discreteModel.actions(state).length() - optimalCount;
    Scalar epsilon = explorationRate.epsilon(state, sac);
    if (nonOptimalCount == 0) {
      Tensor pdf = Tensors.vector(v -> RationalScalar.of(1, optimalCount), bestActions.length());
      return EmpiricalDistribution.fromUnscaledPDF(pdf);
    }
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> index.containsKey(action) //
            ? RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount))
            : epsilon.divide(RealScalar.of(nonOptimalCount))));
    return EmpiricalDistribution.fromUnscaledPDF(pdf);
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    final int nonOptimalCount = discreteModel.actions(state).length() - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    Scalar epsilon = explorationRate.epsilon(state, sac);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
  }

  @Override // from PolicyBase
  public EGreedyPolicy copy() {
    EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(discreteModel, qsa, sac);
    eGreedyPolicy.setExplorationRate(explorationRate);
    return eGreedyPolicy;
  }
}
