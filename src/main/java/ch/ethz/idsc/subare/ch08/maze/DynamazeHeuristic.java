// code by jph
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.nrm.Vector1Norm;
import ch.ethz.idsc.tensor.sca.Power;

enum DynamazeHeuristic {
  ;
  public static DiscreteQsa create(Dynamaze dynamaze) {
    Index terminalIndex = dynamaze.terminalIndex();
    if (terminalIndex.size() != 1)
      throw new RuntimeException("not yet implemented");
    Tensor endPos = terminalIndex.get(0); // for maze2 == {8, 0}
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    for (Tensor key : qsa.keys()) {
      final Tensor state = key.get(0);
      final Tensor action = key.get(1);
      Scalar dist = Vector1Norm.between(state.add(action), endPos);
      // Scalar dist = Norm._1.ofVector(state.subtract(endPos));
      Scalar value = Power.of(dynamaze.gamma(), dist);
      qsa.assign(state, action, value);
    }
    return qsa;
  }

  public static void demo(Dynamaze dynamaze) {
    DiscreteQsa nul = DiscreteQsa.build(dynamaze);
    DiscreteQsa est = create(dynamaze);
    DiscreteQsa qsa = DynamazeHelper.getOptimalQsa(dynamaze);
    System.out.println("diff to zero      = " + DiscreteValueFunctions.distance(qsa, nul));
    System.out.println("diff to heuristic = " + DiscreteValueFunctions.distance(qsa, est));
  }

  public static void main(String[] args) throws Exception {
    demo(DynamazeHelper.original("maze2"));
    demo(DynamazeHelper.create5(2));
  }
}
