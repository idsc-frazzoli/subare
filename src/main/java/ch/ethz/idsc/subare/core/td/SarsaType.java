// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;

public enum SarsaType {
  original, //
  expected, //
  qlearning, //
  ;
  // ---
  public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate //
  ) {
    switch (this) {
    case original:
      return new OriginalSarsa(discreteModel, qsa, learningRate);
    case expected:
      return new ExpectedSarsa(discreteModel, qsa, learningRate);
    case qlearning:
      return new QLearning(discreteModel, qsa, learningRate);
    }
    throw new RuntimeException();
  }
}
