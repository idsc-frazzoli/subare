// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/* package */ interface SarsaEvaluation {
  Scalar evaluate(Scalar epsilon, LearningRate learningRate, Tensor state, QsaInterface qsa);

  Scalar crossEvaluate(Scalar epsilon, Tensor state, Tensor actions, QsaInterface qsa1, QsaInterface qsa2);
}