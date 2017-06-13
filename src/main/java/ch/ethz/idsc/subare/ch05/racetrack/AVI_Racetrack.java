// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Import;

/** action value iteration for gambler's dilemma */
class AVI_Racetrack {
  static void precompute(String name) throws Exception {
    Racetrack racetrack = RacetrackHelper.create(name, 5);
    ActionValueIteration avi = new ActionValueIteration(racetrack);
    avi.untilBelow(RealScalar.of(1e-1), 2);
    Export.object(UserHome.file(name + ".object"), avi.qsa());
  }

  public static void main(String[] args) throws Exception {
    String name = "track2";
    precompute(name);
    DiscreteQsa qsa = Import.object(UserHome.file(name + ".object"));
    System.out.println(qsa.size());
    Racetrack racetrack = RacetrackHelper.create(name, 5);
    int c = 0;
    for (Tensor state : racetrack.states()) {
      for (Tensor action : racetrack.actions(state)) {
        try {
          qsa.value(state, action);
          ++c;
        } catch (Exception exception) {
          // ---
        }
      }
    }
    System.out.println(c + " / " + qsa.size());
    Export.of(UserHome.Pictures("racetrack_qsa_avi_21_11.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(1, 1)));
    Export.of(UserHome.Pictures("racetrack_qsa_avi_21_10.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(1, 0)));
    Export.of(UserHome.Pictures("racetrack_qsa_avi_21_01.png"), //
        RacetrackHelper.render(racetrack, qsa, Tensors.vector(2, 1), Tensors.vector(0, 1)));
  }
}
