package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Abs;

public enum MonteCarloErrorAnalysis {
  LINEAR_QSA() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      GlobalAssert.that(refQsa.size() == currentQsa.size());
      Scalar error = RealScalar.ZERO;
      Scalar delta;
      for (int index = 0; index < refQsa.size(); ++index) {
        delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
        error = error.add(delta);
      }
      return error;
    }
  },
  SQUARE_QSA() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      GlobalAssert.that(refQsa.size() == currentQsa.size());
      Scalar error = RealScalar.ZERO;
      Scalar delta;
      for (int index = 0; index < refQsa.size(); ++index) {
        delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
        error = error.add(Times.of(delta, delta));
      }
      return error;
    }
  },
  LINEAR_POLICY() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      GlobalAssert.that(refQsa.size() == currentQsa.size());
      EGreedyPolicy refPolicy = (EGreedyPolicy) GreedyPolicy.bestEquiprobable(monteCarloInterface, refQsa);
      EGreedyPolicy currentPolicy = (EGreedyPolicy) GreedyPolicy.bestEquiprobable(monteCarloInterface, currentQsa);
      Scalar error = RealScalar.ZERO;
      Scalar delta;
      for (Tensor state : monteCarloInterface.states()) {
        Scalar refValue = refQsa.value(state, refPolicy.getBestActions(state).get(0));
        Tensor currentValues = Tensors.empty();
        currentPolicy.getBestActions(state).forEach(v -> currentValues.append(refQsa.value(state, v)));
        Scalar currentValue = Mean.of(currentValues).Get();
        delta = refValue.subtract(currentValue);
        error = error.add(delta);
      }
      return error;
    }
  },
  SQUARE_POLICY() {
    @Override
    public Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      GlobalAssert.that(refQsa.size() == currentQsa.size());
      EGreedyPolicy refPolicy = (EGreedyPolicy) GreedyPolicy.bestEquiprobable(monteCarloInterface, refQsa);
      EGreedyPolicy currentPolicy = (EGreedyPolicy) GreedyPolicy.bestEquiprobable(monteCarloInterface, currentQsa);
      Scalar error = RealScalar.ZERO;
      Scalar delta;
      for (Tensor state : monteCarloInterface.states()) {
        Scalar refValue = refQsa.value(state, refPolicy.getBestActions(state).get(0));
        Tensor currentValues = Tensors.empty();
        currentPolicy.getBestActions(state).forEach(v -> currentValues.append(refQsa.value(state, v)));
        Scalar currentValue = Mean.of(currentValues).Get();
        delta = refValue.subtract(currentValue);
        error = error.add(Times.of(delta, delta));
      }
      return error;
    }
  },;
  public abstract Scalar getError(MonteCarloInterface monteCarloInterface, DiscreteQsa refQsa, DiscreteQsa currentQsa);
}
