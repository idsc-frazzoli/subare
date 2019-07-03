// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Timing;
import ch.ethz.idsc.tensor.opt.ScalarTensorFunction;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.Round;

public class LearningCompetition {
  private final Map<Point, LearningContender> map = new HashMap<>();
  private final ScalarTensorFunction colorDataFunction = ColorDataGradients.CLASSIC;
  // ---
  private final DiscreteQsa ref;
  private final String name;
  private final Tensor epsilon;
  private final Scalar errorcap;
  private final Scalar errorcap2;
  // ---
  // override default values if necessary:
  public int period = 200;
  public int nstep = 1;
  public int magnify = 5;

  public LearningCompetition(DiscreteQsa ref, String name, Tensor epsilon, Scalar errorcap, Scalar errorcap2) {
    this.ref = ref;
    this.name = name;
    this.epsilon = epsilon.unmodifiable();
    this.errorcap = errorcap;
    this.errorcap2 = errorcap2;
  }

  public void put(Point point, LearningContender learningContender) {
    map.put(point, learningContender);
  }

  private int RESX = 0;

  public void doit() throws Exception {
    RESX = map.keySet().stream().mapToInt(point -> point.x).reduce(Math::max).getAsInt() + 1;
    int RESY = map.keySet().stream().mapToInt(point -> point.y).reduce(Math::max).getAsInt() + 1;
    Tensor image = Array.zeros(RESX + 1 + RESX, RESY, 4);
    try (AnimationWriter animationWriter = AnimationWriter.of(HomeDirectory.Pictures("bulk_" + name + ".gif"), period)) {
      for (int index = 0; index < epsilon.length(); ++index) {
        final int findex = index;
        Timing timing = Timing.started();
        map.entrySet().stream().parallel().forEach(entry -> //
        processEntry(image, entry.getKey(), entry.getValue(), findex));
        System.out.println( //
            String.format("%3d %s sec", index, RealScalar.of(timing.seconds()).map(Round._1)));
        animationWriter.append(ImageResize.nearest(image, magnify));
      }
    }
  }

  private void processEntry(Tensor image, Point point, LearningContender learningContender, int index) {
    ConstantExplorationRate explorationRate = ConstantExplorationRate.of(epsilon.Get(index).number().doubleValue());
    learningContender.stepAndCompare(explorationRate, nstep, ref);
    Infoline infoline = learningContender.infoline(ref);
    {
      Scalar error = infoline.q_error();
      error = Min.of(error.divide(errorcap), RealScalar.ONE);
      image.set(colorDataFunction.apply(error), point.x, point.y);
    }
    {
      Scalar error = infoline.loss();
      error = Min.of(error.divide(errorcap2), RealScalar.ONE);
      image.set(colorDataFunction.apply(error), RESX + 1 + point.x, point.y);
    }
  }
}
