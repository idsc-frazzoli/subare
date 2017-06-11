// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.StateActionRasters;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;

enum GamblerHelper {
  ;
  private static final int MAGNIFY = 2;

  static DiscreteQsa getOptimalQsa(Gambler gambler) {
    ActionValueIteration avi = new ActionValueIteration(gambler);
    avi.untilBelow(DecimalScalar.of(.0001));
    return avi.qsa();
  }

  public static DiscreteVs getOptimalVs(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler, gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return vi.vs();
  }

  public static PolicyInterface getOptimalPolicy(Gambler gambler) {
    ValueIteration vi = new ValueIteration(gambler, gambler);
    vi.untilBelow(RealScalar.of(1e-10));
    return GreedyPolicy.bestEquiprobable(gambler, vi.vs());
  }

  public static Tensor joinAll(Gambler gambler, DiscreteQsa qsa) {
    return ImageResize.of(StateActionRasters.qsaPolicy(new GamblerRaster(gambler), qsa), MAGNIFY);
  }

  public static Tensor joinAll(Gambler gambler, DiscreteQsa qsa, DiscreteQsa ref) {
    return ImageResize.of(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref), MAGNIFY);
  }
}
