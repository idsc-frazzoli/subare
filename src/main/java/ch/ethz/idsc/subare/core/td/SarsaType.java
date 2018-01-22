// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;

public enum SarsaType {
  ORIGINAL() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new OriginalSarsa(discreteModel, qsa, learningRate);
    }
  }, //
  EXPECTED() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new ExpectedSarsa(discreteModel, qsa, learningRate);
    }
  }, //
  QLEARNING() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new QLearning(discreteModel, qsa, learningRate);
    }
  }, //
  ;
  // ---
  public abstract Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate);
}
