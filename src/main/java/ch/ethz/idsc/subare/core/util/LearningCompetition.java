// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.red.Min;

public class LearningCompetition {
  private final Map<Point, LearningContender> map = new HashMap<>();
  private final Interpolation interpolation = Colorscheme.classic();
  private final Tensor BASE = Tensors.vector(255);
  // ---
  private final DiscreteQsa ref;
  private final String name;
  private final Tensor epsilon;
  private final Scalar errorcap;
  // ---
  // override default values if necessary:
  public int PERIOD = 200;
  public int NSTEP = 1;
  public int MAGNIFY = 5;

  public LearningCompetition(DiscreteQsa ref, String name, Tensor epsilon, Scalar errorcap) {
    this.ref = ref;
    this.name = name;
    this.epsilon = epsilon.unmodifiable();
    this.errorcap = errorcap;
  }

  public void put(Point point, LearningContender learningContender) {
    map.put(point, learningContender);
  }

  public void doit() throws Exception {
    int RESX = map.keySet().stream().mapToInt(point -> point.x).reduce(Math::max).getAsInt() + 1;
    int RESY = map.keySet().stream().mapToInt(point -> point.y).reduce(Math::max).getAsInt() + 1;
    Tensor image = Array.zeros(RESX, RESY, 4);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("bulk_" + name + ".gif"), PERIOD);
    for (int index = 0; index < epsilon.length(); ++index) {
      System.out.println(index);
      // TODO can do next loop in parallel
      final int findex = index;
      // for (Entry<Point, LearningContender> entry : )
      map.entrySet().stream().parallel().forEach(entry -> //
      processEntry(image, entry.getKey(), entry.getValue(), findex));
      gsw.append(ImageFormat.of(ImageResize.of(image, MAGNIFY)));
    }
    gsw.close();
  }

  private void processEntry(Tensor image, Point point, LearningContender learningContender, int index) {
    Scalar error = learningContender.stepAndCompare(epsilon.Get(index), NSTEP, ref);
    error = Min.of(error.divide(errorcap), RealScalar.ONE);
    image.set(interpolation.get(BASE.multiply(error)), point.x, point.y);
  }
}